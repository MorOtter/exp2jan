// imports
const express = require("express");
const setPacketArray = require("../public/scripts/packet");
const {
  setUpParticipant,
  addFeedback,
  addAccuracy,
} = require("../controllers/participantDetailsController");

// set router const
const router = express.Router();

router.post("/login", async (req, res) => {
  const packetArray = await setPacketArray();
  req.session.packetArray = packetArray;
  setUpParticipant(req, res);
});

// get feedback view
router.get("/feedback", (req, res) => {
  res.render("feedback.ejs");
});

router.post("/feedback", async (req, res) => {
  addFeedback(req, res);
});

router.post("/accuracy", async (req, res) => {
  addAccuracy(req, res);
});

module.exports = router;
